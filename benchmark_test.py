import cocoex
import datetime
from helper_functions import get_optimizers_safe, run_benchmarks

if __name__ == "__main__":
    
    functions  = list(range(1, 2))
    instances  = list(range(1, 6))
    dimensions = [2]
    
    """
    Example usage, you can tune functions, instances, dimensions however you like as long as theyre inside their respective bounds
    -> functions [1, ..., 24]
    -> instances [1, ..., 110]
    -> dimensions [2, 3, 5, 10, 20, 40]
    """
    
    suite_filter = (
        f"function_indices:{','.join(map(str, functions))} "
        f"instance_indices:{','.join(map(str, instances))} "
        f"dimensions:{','.join(map(str, dimensions))}"
    )

    suite    = cocoex.Suite("bbob", "", suite_filter)
    date_time = datetime.datetime.now()
    observer = cocoex.Observer("bbob", f"result_folder:mealpy_bbob_results_{date_time}")

    optimizers = get_optimizers_safe()
    
    #TO CALL JUST ONE OPTIMIZER YOU NEED TO GIVE IT AS A DICTIONARY:
    run_benchmarks(suite, observer, {"DevBBO": optimizers["DevBBO"]}, "output_single", seed=1,  epoch=200, pop_size=50)
    
    #TO CALL ALL YOU CAN SIMPLY INPUT OPTIMIZERS:
    run_benchmarks(suite, observer, optimizers, "output_all", seed=2, epoch=200, pop_size=50)
    
    # To save coco visualizations:
    # cocopp.main(observer.result_folder)
