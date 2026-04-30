import cocoex
import argparse
from helper_functions import run_benchmarks, get_optimizers_safe, get_suite
from concurrent.futures import ProcessPoolExecutor

def run_benchmarks_all_seeds(function_ids, instance_ids, dimensions, optimizers, out_dir, seeds, epoch, pop_size):
    observer = cocoex.Observer("no_observer", "")
    suite = get_suite(function_ids, instance_ids, dimensions)
    try:
        for seed in seeds:
            run_benchmarks(suite=suite, observer=observer, optimizers=optimizers,
                           out_dir=out_dir, seed=seed, epoch=epoch, pop_size=pop_size)
    except Exception as e:
        print(f"Worker process failed: {e}")
        raise

if __name__ == "__main__":
    optimizers, optimizer_names = get_optimizers_safe(True)
    parser = argparse.ArgumentParser(prog='Mealpy_benchmarks', usage='%(prog)s [options]',
                                     description="The program runs the bbob functions and their instances contained in cocoex bbob suite" \
                                     "on algorithms from the ClustOpt paper, with different seeds.")
    
    parser.add_argument('-p', choices=['f', 'd', 's', 'a', 'i'],
                        help="Parallelization mode: parallelise over (f)unctions, (d)imensions, (s)eeds, (i)nstances or (a)lgorithms")
    parser.add_argument('-f', nargs='+', type=int, choices=range(1, 25),
                        help="Function IDs to benchmark (1-24)")
    parser.add_argument('-d', nargs='+', type=int, choices=[2, 3, 5, 10, 20, 40],
                        help="Dimensions to benchmark in")
    parser.add_argument('-s', nargs='+', type=int,
                        help="Seeds to use")    
    parser.add_argument('-i', nargs='+', type=int, choices=range(1, 111),
                        help="Instance IDs to benchmark (1-110)")
    parser.add_argument('-a', nargs='+', type=str, choices=optimizer_names,
                        help="Mealpy algorithms to benchmark")

    args = parser.parse_args()

    # Defaults
    #BaseDE->OriginalDE, SADE, JADE, SHADE->OriginalSHADE, EnchancedAEO,  
    #ModifiedAEO, OriginalAEO, AugmentedAEO, HI_WOA, OriginalWOA
    #outputs/{algorithm_name}/{problem_id}_{instance_id}/{visualization_type}_{random_seed}.png
    dimensions = [2]
    function_ids = list(range(1, 25))
    instance_ids = list(range(1, 6))
    seeds = [1, 2, 3, 4, 5]
    optimizers_filtered = {"JADE": optimizers["JADE"], "OriginalDE": optimizers["OriginalDE"],
                        "SADE": optimizers["SADE"], "OriginalSHADE": optimizers["OriginalSHADE"],
                        "ModifiedAEO": optimizers["ModifiedAEO"],"OriginalAEO": optimizers["OriginalAEO"], 
                        "AugmentedAEO": optimizers["AugmentedAEO"],"HI_WOA": optimizers["HI_WOA"], 
                        "OriginalWOA": optimizers["OriginalWOA"]}
    
    if args.d: dimensions = args.d
    if args.f: function_ids = args.f
    if args.i: instance_ids = args.i
    if args.s: seeds = args.s
    if args.a: optimizers_filtered = {name: optimizers[name] for name in args.a}

    parallelization = args.p  
    
    #default suite without parallelization
    
    suite = get_suite(function_ids, instance_ids, dimensions)
    observer = cocoex.Observer("no_observer", "")

    if parallelization == 'd':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(dimensions)} dimensions: {dimensions}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for d in dimensions:
                executor.submit(run_benchmarks_all_seeds, function_ids, instance_ids, dimensions, 
                                optimizers_filtered, "outputs_1", seeds, 20, 50)
    elif parallelization == 'f':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(function_ids)} functions: {function_ids}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for f_id in function_ids:
                executor.submit(run_benchmarks_all_seeds, [f_id], instance_ids, dimensions, 
                                optimizers_filtered, "outputs_1", seeds, 20, 50)
    elif parallelization == 'i':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(instance_ids)} instances: {instance_ids}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for i_id in instance_ids:
                executor.submit(run_benchmarks_all_seeds, function_ids, [i_id], dimensions, 
                                optimizers_filtered, "outputs_1", seeds, 20, 50)
    elif parallelization == 'a':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(optimizers_filtered)} algorithms: {list(optimizers_filtered.keys())}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for name, optimizer in optimizers_filtered.items():
                executor.submit(run_benchmarks_all_seeds, function_ids, instance_ids, dimensions, 
                                {name:optimizer}, "outputs_1", seeds, 20, 50)
    elif parallelization == 's':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(seeds)} seeds: {seeds}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for seed in seeds:
                executor.submit(run_benchmarks_all_seeds, function_ids, instance_ids, dimensions, 
                                optimizers_filtered, "outputs_1", [seed], 20, 50)
    else:
        print(f"\n{'='*50}")
        print("Running sequentially (no parallelization)")
        print(f"{'='*50}")
        for seed in seeds:
            run_benchmarks(suite=suite, observer=observer, optimizers=optimizers_filtered,
                        out_dir="outputs_1", seed=seed, epoch=20, pop_size=50)
    
    print(f"{'='*50}")
    print("FINISHED")
