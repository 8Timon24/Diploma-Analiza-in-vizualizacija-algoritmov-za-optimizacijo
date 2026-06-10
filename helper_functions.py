from pathlib import Path
import numpy as np
import csv
import pandas as pd
import os
import mealpy
import inspect
import pkgutil
import importlib
import cocoex

charts = ["save_global_objectives_chart", "save_local_objectives_chart", "save_global_best_fitness_chart", 
            "save_local_best_fitness_chart", "save_runtime_chart", "save_exploration_exploitation_chart",
            "save_diversity_chart", "save_trajectory_chart"]

"""
Generates and saves all built-in Mealpy visualization charts
based on the optimization history of the model. You specifiy
the location of where it saves the .png files with out_dir parameter
and seed is used for naming purposes (the seed you used for the problem).
"""
def generate_all_charts(model, out_dir, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for chart in charts:
        sliced = chart.replace("save_", "").replace("_chart", "")
        getattr(model.history, chart)(title=sliced, filename=str(out_dir / f"{sliced}_{seed}.png"))


"""
Returns a numpy array of x,y and fitness values for the best agent's trajectory in space.
"""
def best_trajectory(model):
    trajectory = []
    for agent in model.history.list_global_best:
        row = list(agent.solution)
        row.append(agent.target.fitness)
        trajectory.append(row)
    return np.array(trajectory)

"""
Saves the best solution found by the algorithm on the given problem instance and seed to a .csv file
"""
def save_best_solution(filename, alg, problem_id, instance_id, seed, x, y):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            coord_cols = [f'x{i+1}' for i in range(len(x))]
            writer.writerow(["algorithm", "problem_id", "instance_id", "seed"] + coord_cols + ["fitness"])
        writer.writerow([alg, problem_id, instance_id, seed] + list(x) + [y])

"""
Saves the trajectory given as an array of 3-tuples to a .csv file.
"""
def save_trajectory(path, traj, dim):
    n_coords = traj.shape[1] - 3  # subtract fitness, iteration, evals
    cols = [f'x{i+1}' for i in range(n_coords)] + ['fitness', 'iteration', 'evaluations']
    pd.DataFrame(traj, columns=cols).to_csv(path, index=False)
""" 
Returns all available optimizers in the current mealpy library version. If verbose=True, it also prints them to the console.
"""
def get_optimizers_safe(verbose=False):
    optimizers = {}
    seen = set()  # Track by (module_path, class_name) to avoid cross-module duplicates

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
            # Deduplicate by identity, not just name
            uid = id(obj)
            if uid in seen:
                continue
            seen.add(uid)

            # Skip non-mealpy classes
            if not (obj.__module__ or "").startswith("mealpy"):
                continue

            # Must have 'solve' to be an optimizer
            if not hasattr(obj, "solve"):
                continue

            # Skip known non-optimizer base classes
            if name in ("Optimizer", "Problem", "Termination"):
                continue

            # ✅ KEY FIX: Always register the class, whether or not it
            # can be instantiated with no args. BaseDE, SHADE, etc. need
            # arguments and were being silently dropped by the try/except.
            optimizers[name] = obj

            if verbose:
                # Optionally test instantiation separately for reporting
                try:
                    obj()
                    print(f"✓ {name} (no-arg instantiable)")
                except Exception:
                    print(f"~ {name} (registered, requires args)")

    return (optimizers, list(optimizers.keys()))

"""
The main function for running benchmarks, suite is the collection of 
functions, dimensions and instances on which we want to run the optimizers on,
optimizers is a dictionary of name:optimizer_object pairs, out_dir is the directory
in which we want to save the results, seed, epoch and pop_size are parameters for running
the algorithm. Example:
-> run_benchmarks(suite, observer, {"DevBBO": optimizers["DevBBO"]}, "output_single", seed=1,  epoch=200, pop_size=50)
-> run_benchmarks(suite, observer, optimizers, "output_all", seed=2, epoch=200, pop_size=50)
"""
def run_benchmarks(suite, observer, optimizers, out_dir, seed=1, epoch=100, pop_size=20, charts=False, results=False):
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
                "save_population": True,
                "seed": seed,
                "log_to": None,
            }

            try:
                model = algo_class(epoch=epoch, pop_size=pop_size)
                result = model.solve(problem_def)
                
                directory = f"{out_dir}/dim_{problem.dimension}/{name}/{problem.id_function}_{problem.id_instance}"
                os.makedirs(directory, exist_ok=True)

                x = model.g_best.solution
                y = model.g_best.target.fitness
                traj = population_trajectory(model, pop_size)
                
                if charts:
                    generate_all_charts(model, directory, seed)
                if results:
                    save_best_solution(f"{out_dir}/results.csv", name, problem.id_function, problem.id_instance, seed, x, y)
                
                save_trajectory(f"{directory}/trajectory_{seed}.csv", traj, problem.dimension)
                
                print(f"  [{name}] {problem.id} | f*={result.target.fitness:.4e} | evals={problem.evaluations}")

            except Exception as e:
                print(f"  [{name}] {problem.id} | FAILED: {e}")

def population_trajectory(model, pop_size):
    trajectory = []
    for iteration, population in enumerate(model.history.list_population):
        evals = (iteration + 1) * pop_size
        for agent in population:
            row = list(agent.solution)  # all dimensions dynamically
            row.append(agent.target.fitness)
            row.append(iteration + 1)
            row.append(evals)
            trajectory.append(row)
    return np.array(trajectory)
"""
Returns a cocoex suite for the parameters which are given as lists of integers:
-> functions [1, ..., 24]
-> instances [1, ..., 110]
-> dimensions [2, 3, 5, 10, 20, 40]
"""
def get_suite(functions, instances, dimensions):
    suite_filter = (
        f"function_indices:{','.join(map(str, functions))} "
        f"instance_indices:{','.join(map(str, instances))} "
        f"dimensions:{','.join(map(str, dimensions))}"
    )

    return cocoex.Suite("bbob", "", suite_filter)
    

