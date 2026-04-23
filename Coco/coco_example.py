import optuna
import optunahub

#function_id = i, ith function from a list of 24, dimension in [2, 3, 5, 10, 20, 40] 
#instance_id = 1 -> the basic version of the function
#instance_id in range [2, 110] different transformations applied to the basic function
bbob = optunahub.load_module("benchmarks/bbob")
sphere2d = bbob.Problem(function_id=10, dimension=2, instance_id=1)

study = optuna.create_study(directions=sphere2d.directions)
study.optimize(sphere2d, n_trials=20)

print(study.best_trial.params, study.best_trial.value)
print(sphere2d.search_space)
