import cocoex
import argparse
from helper_functions import run_benchmarks, get_optimizers_safe, get_suite
from concurrent.futures import ProcessPoolExecutor

def run_benchmarks_all_seeds(function_ids, instance_ids, dimensions, optimizers, out_dir, seeds, epoch_per_dim, pop_size, only_best, save_diversity):
    observer = cocoex.Observer("no_observer", "")
    suite = get_suite(function_ids, instance_ids, dimensions)
    try:
        for seed in seeds:
            run_benchmarks(suite=suite, observer=observer, optimizers=optimizers,
                           out_dir=out_dir, seed=seed, epoch_per_dim=epoch_per_dim, pop_size=pop_size,
                           only_best=only_best, save_diversity=save_diversity)
    except Exception as e:
        print(f"Worker process failed: {e}")
        raise

if __name__ == "__main__":
    optimizers, optimizer_names = get_optimizers_safe()
    ALGORITHMS_OF_INTEREST = ["AugmentedAEO", "GWO_WOA",
                           "HI_WOA", "IGWO",
                           "ImprovedBSO", "JADE",
                           "L_SHADE", "LevyTWO",
                           "ModifiedAEO", "OriginalAEO",
                           "OriginalALO", "OriginalCSA",
                           "OriginalDE", "OriginalFPA",
                           "OriginalGWO", "OriginalHC",
                           "OriginalHHO", "OriginalMFO",
                           "OriginalMPA", "OriginalMRFO",
                           "OriginalNMRA", "OriginalSHADE",
                           "OriginalSSA", "OriginalSSpiderA",
                           "OriginalWOA", "RW_GWO",
                           "SADE", "WhaleFOA"]
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
    parser.add_argument('-b', '--best', action='store_true',
                        help="Only save the best (g_best) trajectory instead of the full population")
    parser.add_argument('-e', '--diversity', action='store_true',
                        help="Also save per-iteration diversity / exploration / exploitation")
    args = parser.parse_args()

    # Defaults
    #BaseDE->OriginalDE, SADE, JADE, SHADE->OriginalSHADE, EnchancedAEO,  
    #ModifiedAEO, OriginalAEO, AugmentedAEO, HI_WOA, OriginalWOA
    #outputs/{algorithm_name}/{problem_id}_{instance_id}/{visualization_type}_{random_seed}.png
    dimensions = [2, 5, 10]
    function_ids = list(range(1, 25))
    instance_ids = list(range(1, 6))
    seeds = [1, 2, 3, 4, 5]
    optimizers_filtered = {name: optimizers[name] for name in ALGORITHMS_OF_INTEREST}

    # epoch = EPOCH_PER_DIM * problem.dimension (ClustOpt convention), computed
    # per-problem inside run_benchmarks - NOT a fixed iteration count shared
    # across all dimensions.
    EPOCH_PER_DIM = 10
    
    if args.d: dimensions = args.d
    if args.f: function_ids = args.f
    if args.i: instance_ids = args.i
    if args.s: seeds = args.s
    if args.a: optimizers_filtered = {name: optimizers[name] for name in args.a}
    
    save_diversity = args.diversity
    best = args.best
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
                executor.submit(run_benchmarks_all_seeds, function_ids, instance_ids, [d], 
                                optimizers_filtered, "outputs", seeds, EPOCH_PER_DIM, 50, best, save_diversity)
    elif parallelization == 'f':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(function_ids)} functions: {function_ids}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for f_id in function_ids:
                executor.submit(run_benchmarks_all_seeds, [f_id], instance_ids, dimensions, 
                                optimizers_filtered, "outputs", seeds, EPOCH_PER_DIM, 50, best, save_diversity)
    elif parallelization == 'i':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(instance_ids)} instances: {instance_ids}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for i_id in instance_ids:
                executor.submit(run_benchmarks_all_seeds, function_ids, [i_id], dimensions, 
                                optimizers_filtered, "outputs", seeds, EPOCH_PER_DIM, 50, best, save_diversity)
    elif parallelization == 'a':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(optimizers_filtered)} algorithms: {list(optimizers_filtered.keys())}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for name, optimizer in optimizers_filtered.items():
                executor.submit(run_benchmarks_all_seeds, function_ids, instance_ids, dimensions, 
                                {name:optimizer}, "outputs", seeds, EPOCH_PER_DIM, 50, best, save_diversity)
    elif parallelization == 's':
        print(f"\n{'='*50}")
        print(f"Parallelizing over {len(seeds)} seeds: {seeds}")
        print(f"{'='*50}")
        with ProcessPoolExecutor() as executor:
            for seed in seeds:
                executor.submit(run_benchmarks_all_seeds, function_ids, instance_ids, dimensions, 
                                optimizers_filtered, "outputs", seeds, EPOCH_PER_DIM, 50, best, save_diversity)
    else:
        print(f"\n{'='*50}")
        print("Running sequentially (no parallelization)")
        print(f"{'='*50}")
        for seed in seeds:
            run_benchmarks(suite=suite, observer=observer, optimizers=optimizers_filtered,
                        out_dir="outputs", seed=seed, epoch_per_dim=EPOCH_PER_DIM, pop_size=50, only_best=best, save_diversity=save_diversity)
    
    print(f"{'='*50}")
    print("FINISHED")