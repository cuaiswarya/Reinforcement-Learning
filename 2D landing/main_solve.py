from pyrlprob.problem import RLProblem
from pyrsistent import b


# Configuration file
config_file = "config_files/landing2d.yaml"


# Define RL problem
LandPrb = RLProblem(config_file)

# Solve RL problem
trainer_dir, exp_dirs, last_cps, best_cp_dir = \
        LandPrb.solve(evaluate=True, postprocess=True, debug=False)

