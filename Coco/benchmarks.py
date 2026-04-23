import mealpy
import numpy as np
import cocoex
import cocopp
import inspect
import datetime

def get_optimizers_safe(verbose=False):
    optimizers = {}

    # MealPy v3+ organizes optimizers in submodules
    # We need to walk submodules, not just the top-level mealpy namespace
    import pkgutil
    import importlib

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=mealpy.__path__,
        prefix=mealpy.__name__ + ".",
        onerror=lambda x: None
    ):
        try:
            module = importlib.import_module(modname)
        except Exception as e:
            if verbose:
                print(f"  Could not import {modname}: {e}")
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip if already found
            if name in optimizers:
                continue

            # Skip the base class itself and non-mealpy classes
            if obj.__module__ and not obj.__module__.startswith("mealpy"):
                continue

            # Try to detect optimizer classes — check for 'solve' method
            if not hasattr(obj, "solve"):
                continue

            # Skip abstract/base optimizer names
            if name in ("Optimizer", "Problem", "Termination"):
                continue

            try:
                instance = obj()
                optimizers[name] = obj
                if verbose:
                    print(f"✓ {name}")
            except Exception as e:
                if verbose:
                    print(f"✗ {name} - SKIPPED: {e}")

    return optimizers


def run_benchmarks(suite, observer, optimizers, epoch=100, pop_size=20):
    for name, algo_class in optimizers.items():
        print(f"\n{'='*50}")
        print(f"Running optimizer: {name}")
        print(f"{'='*50}")

        suite.reset()  

        for problem in suite:
            problem.observe_with(observer)

            def objective(solution, p=problem):
                return p(solution)

            problem_def = {
                "obj_func": objective,
                "bounds": mealpy.FloatVar(
                    lb=list(problem.lower_bounds),
                    ub=list(problem.upper_bounds)
                ),
                "minmax": "min",
                "log_to": None,
            }

            try:
                model = algo_class(epoch=epoch, pop_size=pop_size)
                result = model.solve(problem_def)
                print(f"  [{name}] {problem.id} | f*={result.target.fitness:.4e} | evals={problem.evaluations}")
            except Exception as e:
                print(f"  [{name}] {problem.id} | FAILED: {e}")


if __name__ == "__main__":
    functions  = list(range(1, 2))
    instances  = list(range(1, 6))
    dimensions = [2]

    suite_filter = (
        f"function_indices:{','.join(map(str, functions))} "
        f"instance_indices:{','.join(map(str, instances))} "
        f"dimensions:{','.join(map(str, dimensions))}"
    )

    suite    = cocoex.Suite("bbob", "", suite_filter)
    date_time = datetime.datetime.now()
    observer = cocoex.Observer("bbob", f"result_folder:mealpy_bbob_results_{date_time}")

    optimizers = get_optimizers_safe(verbose=True)
    run_benchmarks(suite, observer, optimizers, epoch=50, pop_size=10)
    cocopp.main(observer.result_folder)
