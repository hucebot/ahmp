import time

import numpy as np

from cem.params import Params

def calc_init_probs(dim_discrete, n_values):
    return [[1.0 / n_values[i] for _ in range(n_values[i])] for i in range(dim_discrete)]

## Chimney Experiment Parametes
CHIMNEY_PARAMS = Params()
CHIMNEY_PARAMS.seed = int(time.time())

CHIMNEY_PARAMS.base_target_xy = [0.0, 0.0]
CHIMNEY_PARAMS.base_target_dz = 3.0
CHIMNEY_PARAMS.max_trajopt_iter = 60
CHIMNEY_PARAMS.dt = 0.05
CHIMNEY_PARAMS.print_level = 1
CHIMNEY_PARAMS.viol_tol = 1e-5

CHIMNEY_PARAMS.parallel = True
CHIMNEY_PARAMS.n_threads = 16
CHIMNEY_PARAMS.cem_iters = 5
CHIMNEY_PARAMS.pop_size = 8
CHIMNEY_PARAMS.n_elites = int(0.5 * CHIMNEY_PARAMS.pop_size)
CHIMNEY_PARAMS.decrease_pop_factor = 1.0
CHIMNEY_PARAMS.fraction_elites_reused = 0.0
CHIMNEY_PARAMS.prob_keep_previous = 0.0
CHIMNEY_PARAMS.beta = 1.0
CHIMNEY_PARAMS.elem_size = 0

CHIMNEY_PARAMS.dim_discrete = 8
CHIMNEY_PARAMS.n_values = [15 for _ in range(CHIMNEY_PARAMS.dim_discrete)]
CHIMNEY_PARAMS.init_probs = calc_init_probs(CHIMNEY_PARAMS.dim_discrete, CHIMNEY_PARAMS.n_values)
CHIMNEY_PARAMS.min_prob = 0.05

CHIMNEY_PARAMS.dim_continuous = CHIMNEY_PARAMS.dim_discrete
CHIMNEY_PARAMS.min_value_continuous = np.full(CHIMNEY_PARAMS.dim_continuous, -1.6)
CHIMNEY_PARAMS.max_value_continuous = np.full(CHIMNEY_PARAMS.dim_continuous, -0.69)
CHIMNEY_PARAMS.init_mu_continuous = np.full(CHIMNEY_PARAMS.dim_continuous, -1.1)
CHIMNEY_PARAMS.init_std_continuous = np.full(CHIMNEY_PARAMS.dim_continuous, 1.0)
CHIMNEY_PARAMS.min_std_continuous = np.full(CHIMNEY_PARAMS.dim_continuous, 1e-3)

CHIMNEY_PARAMS.viol_pen = -10.0
CHIMNEY_PARAMS.scenario = "chimney"
CHIMNEY_PARAMS.robot = "talos"
CHIMNEY_PARAMS.trajopt_type = "whole_body"

CHIMNEY_PARAMS.vault_mu = 0.8
CHIMNEY_PARAMS.vault_z = 0.65
CHIMNEY_PARAMS.vault_x_min = 0.35
CHIMNEY_PARAMS.vault_x_max = 0.55

CHIMNEY_PARAMS.chimney_mu = 0.9
CHIMNEY_PARAMS.chimney_y = 0.50

## Handrails Experiment Parametes
HANDRAILS_PARAMS = Params()
HANDRAILS_PARAMS.seed = int(time.time())

HANDRAILS_PARAMS.base_target_xy = [3.0, 0.0]
HANDRAILS_PARAMS.base_target_dz = 0.0
HANDRAILS_PARAMS.max_trajopt_iter = 50
HANDRAILS_PARAMS.dt = 0.05
HANDRAILS_PARAMS.print_level = 1
HANDRAILS_PARAMS.viol_tol = 1e-5

HANDRAILS_PARAMS.parallel = True
HANDRAILS_PARAMS.n_threads = 16
HANDRAILS_PARAMS.cem_iters = 5
HANDRAILS_PARAMS.pop_size = 8
HANDRAILS_PARAMS.n_elites = int(0.5 * HANDRAILS_PARAMS.pop_size)
HANDRAILS_PARAMS.decrease_pop_factor = 1.0
HANDRAILS_PARAMS.fraction_elites_reused = 0.0
HANDRAILS_PARAMS.prob_keep_previous = 0.0
HANDRAILS_PARAMS.beta = 1.0
HANDRAILS_PARAMS.elem_size = 0

HANDRAILS_PARAMS.dim_discrete = 10
HANDRAILS_PARAMS.n_values = [15 for _ in range(HANDRAILS_PARAMS.dim_discrete)]
HANDRAILS_PARAMS.init_probs = calc_init_probs(HANDRAILS_PARAMS.dim_discrete, HANDRAILS_PARAMS.n_values)
HANDRAILS_PARAMS.min_prob = 0.05

HANDRAILS_PARAMS.dim_continuous = HANDRAILS_PARAMS.dim_discrete
HANDRAILS_PARAMS.min_value_continuous = np.full(HANDRAILS_PARAMS.dim_continuous, -1.6)
HANDRAILS_PARAMS.max_value_continuous = np.full(HANDRAILS_PARAMS.dim_continuous, -0.35)
HANDRAILS_PARAMS.init_mu_continuous = np.full(HANDRAILS_PARAMS.dim_continuous, -0.625)
HANDRAILS_PARAMS.init_std_continuous = np.full(HANDRAILS_PARAMS.dim_continuous, 1.0)
HANDRAILS_PARAMS.min_std_continuous = np.full(HANDRAILS_PARAMS.dim_continuous, 1e-3)

HANDRAILS_PARAMS.viol_pen = -10.0
HANDRAILS_PARAMS.scenario = "handrails"
HANDRAILS_PARAMS.robot = "talos"
HANDRAILS_PARAMS.trajopt_type = "whole_body"

HANDRAILS_PARAMS.rail_mu = 0.8
HANDRAILS_PARAMS.rail_y = 0.50
HANDRAILS_PARAMS.rail_z = 0.9
HANDRAILS_PARAMS.rail_w = 0.02
