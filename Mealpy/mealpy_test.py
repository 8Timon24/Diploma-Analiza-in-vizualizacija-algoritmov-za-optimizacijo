from mealpy import FloatVar, GA, SMA, BRO
import numpy as np

def objective_func(solution):
    return np.sum(solution**2)+1

problem_dict = {
    "obj_func": objective_func,
    "bounds": FloatVar(lb=[-100, ] * 30, ub=[100, ] * 30,),
    "minmax": "min",
}

ga_model = GA.BaseGA(epoch=100, pop_size=50, pc=0.85, pm=0.1)
ga_model.solve(problem_dict, mode="thread")

sma_model = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03, save_population=True)
sma_model.solve(problem_dict)

br_model = BRO.OriginalBRO(epoch=110, pop_size=50)
br_model.solve(problem_dict, mode="swarm")

print("GA solution: ", ga_model.g_best.solution)
print("GA fitness: ", ga_model.g_best.target.fitness)

print("SMA solution: ", sma_model.g_best.solution)
print("SMA fitness: ", sma_model.g_best.target.fitness)

print("br solution: ", br_model.g_best.solution)
print("br fitness: ", br_model.g_best.target.fitness)

sma_model.history.save_diversity_chart()
sma_model.history.save_exploration_exploitation_chart()
