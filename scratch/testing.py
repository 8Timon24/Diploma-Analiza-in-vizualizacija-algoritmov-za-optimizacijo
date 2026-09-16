# Sanity check: compares the known BBOB optimum for one (function, instance,
# dim) against the best fitness harvest_results.py found, to confirm no run
# beat the true optimum (which would indicate a bug).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cocoex as ex
import numpy as np
import os
import pandas as pd
from config import OUTPUTS_DIR

def get_bbob_optimum(function, instance, dim):
    suite_filter = f"function_indices:{function} instance_indices:{instance} dimensions:{dim}"
    suite = ex.Suite("bbob", "", suite_filter)
    problem = suite.next_problem()

    tmpfile = '._bbob_problem_best_parameter.txt'   # NOTE: leading dot!
    problem._best_parameter(what="print")
    x_opt = np.loadtxt(tmpfile)
    os.remove(tmpfile)

    f_opt = problem(x_opt)
    return x_opt, f_opt

if __name__ == '__main__':
    x, f_opt = get_bbob_optimum(21, 1, 2)
    res = pd.read_csv(f'{OUTPUTS_DIR}/dim_2/results.csv')
    best_found = res.query('problem_id==21 and instance_id==1')['fitness'].min()
    print(f"f_opt = {f_opt}, best found = {best_found}")
    print("consistent (no run beat the optimum):", best_found >= f_opt - 1e-6)