"""Benchmark problem sources: BBOB, opfunu, CEC, and user-defined.

helper_functions.run_benchmarks was written against cocoex, but it only ever
touches a small slice of a cocoex problem:

    problem(x), problem.dimension, problem.lower_bounds, problem.upper_bounds,
    problem.id_function, problem.id_instance, problem.id, problem.evaluations,
    problem.observe_with(observer)

and, on the suite, iteration plus reset(). That surface is small enough to
imitate, so every non-BBOB source here is an adapter and run_benchmarks runs
unmodified - there is no second copy of the optimization loop to drift.

Two traps this module exists to absorb:

* opfunu does not raise when a name-based function cannot take the requested
  dimension. Ackley02 asked for ndim=10 prints "is fixed problem with 2
  variables!" and hands back a 2-D problem. The actual dimension is therefore
  always read back off the instance, never assumed from the request.
* A CEC function constructed at an unsupported dimension can abort the
  process outright - F112017(ndim=2) dies with no Python exception, so no
  try/except can contain it. CEC functions are therefore never probed at a
  requested dimension. Each is constructed once at its OWN default, which is
  always safe, and its dim_supported list decides what may be offered.
* preprocess_data.py splits a run directory on "_" and int()s the first part,
  so id_function has to be an integer. Function names are recorded in the run
  manifest instead of being used as directory names.
* COCO's instance_indices filter is POSITIONAL into whatever instance list
  the suite was declared with, and the default bbob suite declares only 15
  instances: 1-5 and 71-80. Asking for "instance 6" against the default
  therefore silently yields instance 71, and anything past 15 is ignored so
  the filter returns every instance instead of one. BBOBSource declares
  instances:1-110 so that position N really is instance N.
"""
import io
import math
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

BBOB_KEY = "bbob"
OPFUNU_KEY = "opfunu"
CEC_KEY = "cec"
CUSTOM_KEY = "custom"


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------

class AdaptedProblem:
    """Duck-types the part of a cocoex problem that run_benchmarks uses."""

    def __init__(self, evaluate, lower_bounds, upper_bounds, id_function,
                 id_instance=1, name="", source=""):
        self._evaluate = evaluate
        self.lower_bounds = list(lower_bounds)
        self.upper_bounds = list(upper_bounds)
        self.dimension = len(self.lower_bounds)
        self.id_function = int(id_function)
        self.id_instance = int(id_instance)
        self.name = name
        self.source = source
        self.evaluations = 0

    @property
    def id(self):
        return (
            f"{self.source}_f{self.id_function:03d}"
            f"_i{self.id_instance:02d}_d{self.dimension:02d}"
        )

    def observe_with(self, observer):
        """cocoex hook. Nothing observes these problems; COCO's own observer
        is a no-op in this project too ("no_observer")."""
        return self

    def reset(self):
        self.evaluations = 0

    def __call__(self, x):
        self.evaluations += 1
        value = float(self._evaluate(np.asarray(x, dtype=float)))
        # mealpy minimises a float; a NaN would silently poison the run.
        return value if math.isfinite(value) else float("inf")


class AdaptedSuite:
    """Duck-types cocoex.Suite: iterable, resettable."""

    def __init__(self, problems):
        self._problems = list(problems)

    def __iter__(self):
        return iter(self._problems)

    def __len__(self):
        return len(self._problems)

    def reset(self):
        for problem in self._problems:
            problem.reset()


@dataclass(frozen=True)
class FunctionSpec:
    """One selectable benchmark function."""

    id: int
    name: str
    label: str
    native_dimension: int = None   # set when the function's dimension is fixed

    @property
    def is_fixed_dimension(self):
        return self.native_dimension is not None

    def describe(self):
        if self.is_fixed_dimension:
            return f"{self.label}  (fixed at {self.native_dimension}-D)"
        return self.label


# --------------------------------------------------------------------------
# opfunu helpers
# --------------------------------------------------------------------------

def _instantiate_quietly(cls, ndim=None):
    """opfunu prints to stdout when it clamps a fixed-dimension function.

    ndim=None constructs at the function's own default, which every opfunu
    function accepts.
    """
    with redirect_stdout(io.StringIO()):
        return cls() if ndim is None else cls(ndim=ndim)


@lru_cache(maxsize=1)
def _name_based_classes():
    import opfunu

    # get_functions_based_ndim() mixes CEC functions in with the name-based
    # ones, so the deduplicated name-based list is the right catalog here.
    return sorted(opfunu.get_all_name_based_functions(), key=lambda c: c.__name__)


@lru_cache(maxsize=1)
def _cec_classes_by_year():
    import opfunu

    by_year = {}
    for cls in opfunu.get_all_cec_based_functions():
        year = cls.__module__.split(".")[-1]
        by_year.setdefault(year, []).append(cls)
    return {
        year: sorted(classes, key=lambda c: c.__name__)
        for year, classes in sorted(by_year.items())
    }


def cec_years():
    return list(_cec_classes_by_year())


@lru_cache(maxsize=1)
def _cec_supported_dimensions():
    """class name -> dimensions that class accepts.

    Built by constructing every CEC function at its own default dimension.
    That is the only safe probe: constructing one at a dimension it does not
    support can kill the interpreter (see the module docstring), so the
    supported list must be discovered without ever guessing a dimension.
    """
    supported = {}
    for classes in _cec_classes_by_year().values():
        for cls in classes:
            try:
                instance = _instantiate_quietly(cls)
            except Exception:
                supported[cls.__name__] = ()
                continue
            dims = getattr(instance, "dim_supported", None) or [
                getattr(instance, "ndim", None)
            ]
            supported[cls.__name__] = tuple(d for d in dims if d)
    return supported


def _spec_for(cls, index, ndim):
    """A FunctionSpec, or None if this function cannot take this dimension."""
    try:
        instance = _instantiate_quietly(cls, ndim)
    except Exception:
        return None
    actual = len(instance.lb)
    label = getattr(instance, "name", cls.__name__) or cls.__name__
    if actual != ndim:
        # Fixed-dimension function: opfunu clamped it rather than refusing.
        return FunctionSpec(id=index, name=cls.__name__, label=label,
                            native_dimension=actual)
    return FunctionSpec(id=index, name=cls.__name__, label=label)


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

class ProblemSource:
    key = ""
    label = ""
    description = ""
    supports_instances = False

    def instance_choices(self):
        """Instance numbers this source offers. Only BBOB has instances."""
        return [1]

    def catalog(self, dimension):
        """Functions selectable at this dimension, as FunctionSpecs."""
        raise NotImplementedError

    def build(self, function_ids, instances, dimensions):
        """A suite run_benchmarks can consume."""
        raise NotImplementedError

    def catalog_union(self, dimensions):
        """[(FunctionSpec, dimensions_it_can_run_at)] for every function
        runnable at AT LEAST ONE of the given dimensions.

        Union rather than intersection on purpose. Most opfunu functions are
        fixed at a single dimension - 65 of the 125 name-based ones are 2-D
        only - so intersecting across a multi-dimension selection would hide
        the majority of the library the moment a second dimension is ticked.
        A function simply runs at the dimensions it supports and is skipped
        at the others.
        """
        by_id = {}
        for dimension in dimensions:
            for spec in self.catalog(dimension):
                entry = by_id.setdefault(spec.id, [spec, []])
                entry[1].append(dimension)
        return [
            (spec, tuple(dims))
            for spec, dims in sorted(by_id.values(), key=lambda e: e[0].id)
        ]

    def count_problems(self, function_ids, instances, dimensions):
        """How many (function, instance, dimension) problems this really is.

        Not the plain product: a function fixed at one dimension contributes
        only at that dimension, so the naive product would overstate the cost.
        """
        wanted = set(int(f) for f in function_ids)
        pairs = sum(
            len(dims) for spec, dims in self.catalog_union(dimensions)
            if spec.id in wanted
        )
        return pairs * max(len(instances), 1) if self.supports_instances else pairs


class BBOBSource(ProblemSource):
    key = BBOB_KEY
    label = "BBOB (COCO)"
    description = (
        "The 24 noiseless BBOB functions. COCO defines 110 instances of each; "
        "the stored results in this repo use instances 1-5."
    )
    supports_instances = True

    # helper_functions.get_suite declares instances:1-110, which is what
    # makes instance_indices mean the instance number rather than a position
    # in COCO's 15-entry default list. Building the suite is delegated to it
    # rather than duplicated here, so the two cannot drift.
    MAX_INSTANCE = 110

    def instance_choices(self):
        return list(range(1, self.MAX_INSTANCE + 1))

    def catalog(self, dimension):
        return [
            FunctionSpec(id=f, name=f"F{f}", label=f"F{f}")
            for f in config.FUNCTIONS
        ]

    def build(self, function_ids, instances, dimensions):
        import helper_functions as hf

        return hf.get_suite(list(function_ids), list(instances), list(dimensions))

    def catalog_union(self, dimensions):
        specs = self.catalog(dimensions[0] if dimensions else 2)
        return [(spec, tuple(dimensions)) for spec in specs]

    def count_problems(self, function_ids, instances, dimensions):
        return len(list(function_ids)) * len(list(instances)) * len(list(dimensions))


class _OpfunuSourceBase(ProblemSource):
    supports_instances = False

    # Building a catalog instantiates every function in the source (~0.15s for
    # the 125 name-based ones). The UI rebuilds it whenever the dimension
    # selection changes, so it is cached per (source, dimension).
    _catalog_cache = {}

    def _classes(self):
        raise NotImplementedError

    def _build_catalog(self, dimension, include_fixed):
        key = (self.key, dimension, include_fixed)
        if key not in self._catalog_cache:
            specs = []
            for index, cls in enumerate(self._classes(), start=1):
                spec = _spec_for(cls, index, dimension)
                if spec is None:
                    continue
                if spec.is_fixed_dimension and not include_fixed:
                    continue
                specs.append(spec)
            self._catalog_cache[key] = tuple(specs)
        return list(self._catalog_cache[key])

    def catalog(self, dimension):
        return self._build_catalog(dimension, include_fixed=False)

    def full_catalog(self, dimension):
        """Every function, including ones fixed at another dimension, so the
        UI can explain an omission rather than silently making one."""
        return self._build_catalog(dimension, include_fixed=True)

    def build(self, function_ids, instances, dimensions):
        classes = self._classes()
        wanted = set(int(f) for f in function_ids)
        problems = []
        for dimension in dimensions:
            for index, cls in enumerate(classes, start=1):
                if index not in wanted:
                    continue
                instance = _instantiate_quietly(cls, dimension)
                if len(instance.lb) != dimension:
                    # Fixed-dimension function asked for a size it cannot do.
                    continue
                problems.append(
                    AdaptedProblem(
                        evaluate=instance.evaluate,
                        lower_bounds=instance.lb,
                        upper_bounds=instance.ub,
                        id_function=index,
                        id_instance=1,
                        name=getattr(instance, "name", cls.__name__),
                        source=self.key,
                    )
                )
        return AdaptedSuite(problems)

    def function_names(self, function_ids):
        """id -> class name, for the run manifest. Directory names are
        integers, so this mapping is the only record of what F7 meant."""
        classes = self._classes()
        wanted = set(int(f) for f in function_ids)
        return {
            index: cls.__name__
            for index, cls in enumerate(classes, start=1)
            if index in wanted
        }


class OpfunuSource(_OpfunuSourceBase):
    key = OPFUNU_KEY
    label = "opfunu (classic functions)"
    description = (
        "Classic benchmark functions - Ackley, Rastrigin, Rosenbrock, "
        "Schwefel and friends - with known optima. No instances."
    )

    def _classes(self):
        return _name_based_classes()


class CECSource(_OpfunuSourceBase):
    key = CEC_KEY
    label = "CEC competition suites"
    description = "The CEC competition benchmark suites, by year."

    def __init__(self, year="cec2017"):
        self.year = year

    @property
    def label_with_year(self):
        return f"CEC {self.year.replace('cec', '')}"

    def _classes(self):
        return _cec_classes_by_year().get(self.year, [])

    @property
    def key(self):
        return f"{CEC_KEY}_{self.year}"

    def supported_dimensions(self):
        """Every dimension any function in this suite accepts."""
        supported = _cec_supported_dimensions()
        dims = set()
        for cls in self._classes():
            dims.update(supported.get(cls.__name__, ()))
        return sorted(dims)

    def _build_catalog(self, dimension, include_fixed):
        key = (self.key, dimension, include_fixed)
        if key not in self._catalog_cache:
            supported = _cec_supported_dimensions()
            specs = []
            for index, cls in enumerate(self._classes(), start=1):
                dims = supported.get(cls.__name__, ())
                takes_it = dimension in dims
                if not takes_it and not include_fixed:
                    continue
                specs.append(
                    FunctionSpec(
                        id=index,
                        name=cls.__name__,
                        label=cls.__name__,
                        native_dimension=None if takes_it else (dims[0] if dims else None),
                    )
                )
            self._catalog_cache[key] = tuple(specs)
        return list(self._catalog_cache[key])

    def build(self, function_ids, instances, dimensions):
        supported = _cec_supported_dimensions()
        classes = self._classes()
        wanted = set(int(f) for f in function_ids)
        problems = []
        for dimension in dimensions:
            for index, cls in enumerate(classes, start=1):
                if index not in wanted:
                    continue
                # Never construct at an unsupported dimension: that is the
                # call that can take the whole process down.
                if dimension not in supported.get(cls.__name__, ()):
                    continue
                instance = _instantiate_quietly(cls, dimension)
                problems.append(
                    AdaptedProblem(
                        evaluate=instance.evaluate,
                        lower_bounds=instance.lb,
                        upper_bounds=instance.ub,
                        id_function=index,
                        id_instance=1,
                        name=getattr(instance, "name", cls.__name__),
                        source=self.key,
                    )
                )
        return AdaptedSuite(problems)


# --------------------------------------------------------------------------
# user-defined functions
# --------------------------------------------------------------------------

# A deliberately small numpy-only namespace. This is the user's own machine
# and their own expression, so the bar is "no accidental foot-guns" - no
# builtins, no imports, no attribute access to anything but these - rather
# than adversarial isolation.
SAFE_NAMES = {
    name: getattr(np, name)
    for name in (
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh",
        "tanh", "exp", "log", "log2", "log10", "sqrt", "abs", "sign",
        "floor", "ceil", "round", "sum", "prod", "mean", "min", "max",
        "power", "maximum", "minimum", "clip", "pi", "e", "arange", "where",
    )
}


class ExpressionError(ValueError):
    """The user's expression could not be compiled or evaluated."""


class CustomSource(ProblemSource):
    key = CUSTOM_KEY
    label = "User-defined function"
    description = (
        "A Python expression in x (a numpy array of the decision variables) "
        "evaluated with numpy only - no builtins and no imports."
    )
    supports_instances = False

    def __init__(self, expression, lower, upper, name="Custom"):
        self.expression = expression
        self.lower = float(lower)
        self.upper = float(upper)
        self.name = name
        try:
            self._code = compile(expression, "<user function>", "eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Could not parse the expression: {exc.msg}") from exc

    def evaluate(self, x):
        try:
            value = eval(self._code, {"__builtins__": {}}, {"x": x, **SAFE_NAMES})
        except Exception as exc:
            raise ExpressionError(f"{type(exc).__name__}: {exc}") from exc
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ExpressionError(
                f"The expression produced {type(value).__name__}, not a number. "
                "It must reduce x to a single value - sum(x**2), not x**2."
            ) from exc

    def validate(self, dimension):
        """Evaluate once at the centre of the box, before a run starts.

        Catching a typo here costs milliseconds; catching it inside
        run_benchmarks means every problem fails one at a time.
        """
        midpoint = np.full(dimension, (self.lower + self.upper) / 2.0)
        value = self.evaluate(midpoint)
        if not math.isfinite(value):
            raise ExpressionError(
                f"The expression is not finite at the centre of the box "
                f"(got {value}). Check for division by zero or log of a "
                f"non-positive number."
            )
        return value

    def catalog(self, dimension):
        return [FunctionSpec(id=1, name=self.name, label=self.name)]

    def build(self, function_ids, instances, dimensions):
        problems = []
        for dimension in dimensions:
            self.validate(dimension)
            problems.append(
                AdaptedProblem(
                    evaluate=self.evaluate,
                    lower_bounds=[self.lower] * dimension,
                    upper_bounds=[self.upper] * dimension,
                    id_function=1,
                    id_instance=1,
                    name=self.name,
                    source=self.key,
                )
            )
        return AdaptedSuite(problems)

    def catalog_union(self, dimensions):
        return [(self.catalog(dimensions[0] if dimensions else 2)[0], tuple(dimensions))]

    def count_problems(self, function_ids, instances, dimensions):
        return len(list(dimensions))

    def function_names(self, function_ids):
        return {1: self.name}


# --------------------------------------------------------------------------

def source_from_config(cfg):
    """Rebuild a source from the dict stored on a RunSpec (and its manifest)."""
    cfg = cfg or {}
    kind = cfg.get("kind", BBOB_KEY)
    if kind == BBOB_KEY:
        return BBOBSource()
    if kind == OPFUNU_KEY:
        return OpfunuSource()
    if kind == CEC_KEY:
        return CECSource(cfg.get("year", "cec2017"))
    if kind == CUSTOM_KEY:
        return CustomSource(
            cfg["expression"], cfg["lower"], cfg["upper"],
            name=cfg.get("name", "Custom"),
        )
    raise ValueError(f"Unknown problem source: {kind!r}")


def available_sources():
    """Every source, in the order the UI should offer them."""
    sources = [BBOBSource(), OpfunuSource()]
    sources.extend(CECSource(year) for year in cec_years())
    sources.append(CustomSource("sum(x**2)", -5.0, 5.0, name="Sphere"))
    return sources
