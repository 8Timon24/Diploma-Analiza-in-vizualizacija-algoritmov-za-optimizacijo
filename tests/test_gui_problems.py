# Unit tests for gui.core.problems: the adapters that let non-BBOB benchmark
# suites drive helper_functions.run_benchmarks unmodified.
#
# Two hazards are pinned here, both found by running the real libraries:
#
#  * opfunu silently CLAMPS a fixed-dimension name-based function instead of
#    refusing it - Ackley02 asked for ndim=10 returns a 2-D problem. Trusting
#    the requested dimension would produce mismatched bounds and wrong output
#    directories.
#  * a CEC function constructed at an unsupported dimension can ABORT the
#    interpreter - F112017(ndim=2) dies with no Python exception, so no
#    try/except can contain it. CEC dimensions must come from dim_supported,
#    never from probing.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from gui.core import problems as P


# -- the interface run_benchmarks depends on ---------------------------

# Exactly what helper_functions.run_benchmarks touches on a problem. If this
# list and the adapter ever drift, every non-BBOB run breaks at once.
REQUIRED_ATTRIBUTES = (
    "dimension", "lower_bounds", "upper_bounds",
    "id_function", "id_instance", "id", "evaluations", "observe_with",
)


def make_problem(evaluate=None, dimension=3):
    return P.AdaptedProblem(
        evaluate=evaluate or (lambda x: float(np.sum(x ** 2))),
        lower_bounds=[-1.0] * dimension,
        upper_bounds=[1.0] * dimension,
        id_function=7,
        id_instance=1,
        name="Test",
        source="test",
    )


@pytest.mark.parametrize("attribute", REQUIRED_ATTRIBUTES)
def test_adapter_exposes_everything_run_benchmarks_uses(attribute):
    assert hasattr(make_problem(), attribute)


def test_adapter_is_callable_like_a_cocoex_problem():
    problem = make_problem()
    assert problem([1.0, 2.0, 2.0]) == pytest.approx(9.0)


def test_dimension_follows_the_bounds_not_a_request():
    assert make_problem(dimension=5).dimension == 5


def test_evaluations_are_counted():
    problem = make_problem()
    assert problem.evaluations == 0
    for _ in range(3):
        problem([0.0, 0.0, 0.0])
    assert problem.evaluations == 3


def test_reset_clears_the_evaluation_count():
    problem = make_problem()
    problem([0.0, 0.0, 0.0])
    problem.reset()
    assert problem.evaluations == 0


def test_non_finite_values_become_infinity():
    # mealpy minimises a float; a NaN would silently poison the comparison
    # and the run would "succeed" with meaningless results.
    problem = make_problem(evaluate=lambda x: float("nan"))
    assert problem([0.0, 0.0, 0.0]) == float("inf")


def test_id_function_is_an_integer():
    # preprocess_data.py does problem_folder.split("_") then int(...), so a
    # non-integer id would break preprocessing of any GUI run.
    assert isinstance(make_problem().id_function, int)
    assert isinstance(make_problem().id_instance, int)


def test_observe_with_is_a_harmless_no_op():
    assert make_problem().observe_with(object()) is not None


def test_suite_can_be_iterated_more_than_once():
    # run_benchmarks iterates the suite once per optimizer.
    suite = P.AdaptedSuite([make_problem(), make_problem()])
    assert len(list(suite)) == 2
    assert len(list(suite)) == 2


def test_suite_reset_clears_every_problem():
    problems = [make_problem(), make_problem()]
    suite = P.AdaptedSuite(problems)
    for problem in problems:
        problem([0.0, 0.0, 0.0])
    suite.reset()
    assert all(problem.evaluations == 0 for problem in problems)


# -- user-defined functions ---------------------------------------------

def test_a_valid_expression_evaluates():
    source = P.CustomSource("sum(x**2)", -1, 1)
    assert source.validate(4) == pytest.approx(0.0)


def test_expression_must_reduce_to_a_scalar():
    # "x**2" is the obvious mistake: it returns an array, not a fitness.
    with pytest.raises(P.ExpressionError, match="not a number"):
        P.CustomSource("x**2", -1, 1).validate(3)


def test_syntax_errors_are_reported_before_any_run():
    with pytest.raises(P.ExpressionError, match="parse"):
        P.CustomSource("sum(x**", -1, 1)


def test_non_finite_expressions_are_rejected_up_front():
    with pytest.raises(P.ExpressionError, match="not finite"):
        P.CustomSource("log(sum(x) - sum(x))", -1, 1).validate(3)


@pytest.mark.parametrize("expression", [
    "__import__('os').listdir('.')",
    "open('/etc/passwd').read()",
    "eval('1+1')",
])
def test_builtins_are_unavailable_to_user_expressions(expression):
    with pytest.raises(P.ExpressionError):
        P.CustomSource(expression, -1, 1).validate(2)


def test_numpy_helpers_are_available():
    source = P.CustomSource("sum(x**2) - 10*sum(cos(2*pi*x)) + 10*3", -5, 5)
    assert isinstance(source.validate(3), float)


def test_custom_source_builds_one_problem_per_dimension():
    suite = P.CustomSource("sum(x**2)", -2, 2).build([1], [1], [2, 5])
    dimensions = sorted(problem.dimension for problem in suite)
    assert dimensions == [2, 5]


# -- source configuration round-trip ------------------------------------

@pytest.mark.parametrize("cfg,expected_key", [
    ({"kind": "bbob"}, "bbob"),
    ({"kind": "opfunu"}, "opfunu"),
    ({"kind": "cec", "year": "cec2014"}, "cec_cec2014"),
])
def test_sources_rebuild_from_their_stored_config(cfg, expected_key):
    assert P.source_from_config(cfg).key == expected_key


def test_custom_source_rebuilds_from_config():
    source = P.source_from_config(
        {"kind": "custom", "expression": "sum(abs(x))", "lower": -3, "upper": 3}
    )
    assert source.validate(2) == pytest.approx(0.0)


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown problem source"):
        P.source_from_config({"kind": "nope"})


# -- opfunu / CEC catalogs ----------------------------------------------

def test_fixed_dimension_functions_are_excluded_not_clamped():
    opfunu = pytest.importorskip("opfunu") and P.OpfunuSource()
    at_two = {spec.name for spec in opfunu.catalog(2)}
    at_ten = {spec.name for spec in opfunu.catalog(10)}
    # Ackley02 is fixed at 2-D: usable at 2, and must not appear at 10.
    assert "Ackley02" in at_two
    assert "Ackley02" not in at_ten


def test_building_a_fixed_dimension_function_at_the_wrong_size_yields_nothing():
    pytest.importorskip("opfunu")
    source = P.OpfunuSource()
    ackley02 = next(s for s in source.full_catalog(10) if s.name == "Ackley02")
    assert ackley02.is_fixed_dimension
    assert len(source.build([ackley02.id], [1], [10])) == 0


def test_every_built_problem_has_the_dimension_that_was_asked_for():
    pytest.importorskip("opfunu")
    source = P.OpfunuSource()
    ids = [spec.id for spec in source.catalog(5)[:5]]
    suite = source.build(ids, [1], [5])
    assert suite and all(problem.dimension == 5 for problem in suite)


def test_cec_catalog_uses_declared_support_not_probing():
    # F112017 aborts the interpreter when constructed at ndim=2. Reaching
    # this assertion at all is the real test; a regression here would not
    # fail, it would kill the test run.
    pytest.importorskip("opfunu")
    source = P.CECSource("cec2017")
    names_at_two = {spec.name for spec in source.catalog(2)}
    assert "F112017" not in names_at_two
    assert "F112017" in {spec.name for spec in source.catalog(10)}


def test_cec_suite_reports_the_dimensions_it_is_defined_for():
    pytest.importorskip("opfunu")
    assert P.CECSource("cec2017").supported_dimensions() == [2, 10, 20, 30, 50, 100]
    # cec2010 is a large-scale suite: nothing at small dimensions.
    assert P.CECSource("cec2010").catalog(10) == []


def test_function_names_map_ids_back_to_names_for_the_manifest():
    pytest.importorskip("opfunu")
    source = P.OpfunuSource()
    names = source.function_names([1, 2])
    assert set(names) == {1, 2}
    assert all(isinstance(name, str) for name in names.values())


# -- union across dimensions --------------------------------------------
#
# Most opfunu name-based functions are fixed at a single dimension - 65 of
# 125 are 2-D only - so a multi-dimension selection must offer the union and
# run each function only where it fits. Intersecting would hide the bulk of
# the library the moment a second dimension is ticked.

def test_union_keeps_functions_that_only_fit_one_of_the_dimensions():
    pytest.importorskip("opfunu")
    source = P.OpfunuSource()
    union = {spec.name: dims for spec, dims in source.catalog_union([2, 10])}
    # Ackley02 is 2-D only, Ackley01 is dimension-changeable: both offered.
    assert union["Ackley02"] == (2,)
    assert set(union["Ackley01"]) == {2, 10}


def test_union_is_at_least_as_large_as_any_single_dimension():
    pytest.importorskip("opfunu")
    source = P.OpfunuSource()
    union = source.catalog_union([2, 10])
    assert len(union) >= len(source.catalog(2))
    assert len(union) >= len(source.catalog(10))


def test_problem_count_skips_dimensions_a_function_cannot_run_at():
    pytest.importorskip("opfunu")
    source = P.OpfunuSource()
    union = dict(
        (spec.name, spec.id) for spec, _dims in source.catalog_union([2, 10])
    )
    changeable, fixed = union["Ackley01"], union["Ackley02"]
    # 2 functions x 2 dimensions would be 4, but Ackley02 only runs at 2.
    assert source.count_problems([changeable, fixed], [1], [2, 10]) == 3


def test_build_runs_each_function_only_where_it_fits():
    pytest.importorskip("opfunu")
    source = P.OpfunuSource()
    union = dict(
        (spec.name, spec.id) for spec, _dims in source.catalog_union([2, 10])
    )
    suite = source.build([union["Ackley01"], union["Ackley02"]], [1], [2, 10])
    built = sorted((p.id_function, p.dimension) for p in suite)
    assert built == [
        (union["Ackley01"], 2), (union["Ackley01"], 10), (union["Ackley02"], 2),
    ] or built == sorted([
        (union["Ackley01"], 2), (union["Ackley01"], 10), (union["Ackley02"], 2),
    ])


def test_bbob_counts_the_plain_product():
    assert P.BBOBSource().count_problems([1, 2, 3], [1, 2], [2, 5, 10]) == 18


def test_custom_counts_one_problem_per_dimension():
    source = P.CustomSource("sum(x**2)", -1, 1)
    assert source.count_problems([1], [1], [2, 5, 10]) == 3


def test_a_dimension_that_makes_opfunu_raise_is_survived():
    # At ndim=100 at least one opfunu function raises IndexError while being
    # constructed. That must shrink the catalog, not take the app down.
    pytest.importorskip("opfunu")
    assert len(P.OpfunuSource().catalog(100)) > 0


# -- BBOB instances -----------------------------------------------------
#
# COCO's instance_indices filter is positional into the instance list the
# suite was DECLARED with. The default bbob suite declares 15 instances
# (1-5 and 71-80), so against it "instance 6" silently means instance 71 and
# anything past 15 is ignored, returning every instance instead of one.
# BBOBSource declares instances:1-110 so the index and the instance number
# are the same thing.

def test_bbob_offers_all_110_instances():
    assert P.BBOBSource().instance_choices() == list(range(1, 111))


def test_only_bbob_has_instances():
    assert P.OpfunuSource().instance_choices() == [1]
    assert P.CECSource("cec2017").instance_choices() == [1]
    assert P.CustomSource("sum(x**2)", -1, 1).instance_choices() == [1]


def test_requested_instance_number_is_the_instance_you_get():
    # Against the default suite, instance 6 comes back as 71. This is the
    # regression guard for that.
    pytest.importorskip("cocoex")
    suite = P.BBOBSource().build([1], [6], [2])
    assert [problem.id_instance for problem in suite] == [6]


def test_high_instance_numbers_are_reachable():
    pytest.importorskip("cocoex")
    suite = P.BBOBSource().build([1], [42, 110], [2])
    assert sorted(problem.id_instance for problem in suite) == [42, 110]


def test_instances_one_to_five_are_unchanged():
    # The stored results use 1-5; widening the declared instance list must
    # not move them, or GUI runs would stop being comparable with outputs/.
    pytest.importorskip("cocoex")
    suite = P.BBOBSource().build([1], [1, 2, 3, 4, 5], [2])
    assert sorted(problem.id_instance for problem in suite) == [1, 2, 3, 4, 5]


def test_bbob_problem_count_uses_instances():
    assert P.BBOBSource().count_problems([1, 2], [1, 2, 3], [2]) == 6
