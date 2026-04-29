import cocoex
from helper_functions import run_benchmarks, get_optimizers_safe, get_suite

if __name__ == "__main__":
    dimensions = [2]
    function_ids = list(range(1, 2))
    instance_ids = list(range(1, 2))
    seeds = [1, 2, 3, 4, 5]
    #BaseDE, SADE, JADE, SHADE, EnchancedAEO, "SHADE": optimizers["SHADE"], 
    #ModifiedAEO, OriginalAEO, AugmentedAEO, HI_WOA, OriginalWOA
    #outputs/{algorithm_name}/{problem_id}_{instance_id}/{visualization_type}_{random_seed}.png

    optimizers = get_optimizers_safe(True)
    optimizers_filtered = {"JADE": optimizers["JADE"], "OriginalDE": optimizers["OriginalDE"],
                        "SADE": optimizers["SADE"], "OriginalSHADE": optimizers["OriginalSHADE"],
                        "ModifiedAEO": optimizers["ModifiedAEO"],"OriginalAEO": optimizers["OriginalAEO"], 
                        "AugmentedAEO": optimizers["AugmentedAEO"],"HI_WOA": optimizers["HI_WOA"], 
                        "OriginalWOA": optimizers["OriginalWOA"],}

    suite = get_suite(function_ids, instance_ids, dimensions)
    observer = cocoex.Observer("bbob", f"result_folder:outputs")

    for j in range(1, 6):
        for i in seeds: 
            run_benchmarks(suite=suite, observer=observer, optimizers=optimizers_filtered, out_dir=f"outputs_{j}", seed=i, epoch=200, pop_size=50)


