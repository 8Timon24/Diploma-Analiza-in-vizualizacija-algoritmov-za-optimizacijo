
if __name__ == "__main__":
    # Quick folder save test — no benchmarking
    from helper_functions import gen_all_charts
    import mealpy
    from mealpy.evolutionary_based.DE import OriginalDE

    problem_def = {
        "obj_func": lambda x: sum(x**2),
        "bounds": mealpy.FloatVar(lb=[-5]*2, ub=[5]*2),
        "minmax": "min",
        "save_population": True,
        "log_to": None,
        "seed": 1,
    }

    model = OriginalDE(epoch=20, pop_size=10)
    model.solve(problem_def)

    #Mock params
    fake_fid = 1
    fake_iid = 3
    directory = f"output/OriginalDE/{fake_fid}_{fake_iid}"
    gen_all_charts(model, directory, seed=1)
    print(f"Check folder: {directory}")