from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass

import numpy as np

@dataclass
class Params:
    seed: int = None

    # Trajopt parameters
    base_target_xy: List[float] = None
    base_target_dz: float = None
    max_trajopt_iter: int = None
    dt: float = None
    print_level: int = None
    viol_tol: float = None

    # CEM parameters
    parallel: bool = None
    n_threads: int = None
    cem_iters: int = None
    pop_size: int = None
    n_elites: int = None
    decrease_pop_factor: float = None
    fraction_elites_reused: float = None
    prob_keep_previous: float = None
    beta: float = None
    elem_size: int = None

    # Discrete
    dim_discrete: int = None
    n_values: List[int] = None
    init_probs: np.ndarray = None
    min_prob: float = None

    # Continuous
    dim_continuous: int = None
    min_value_continuous: np.ndarray = None
    max_value_continuous: np.ndarray = None
    init_mu_continuous: np.ndarray = None
    init_std_continuous: np.ndarray = None
    min_std_continuous: np.ndarray = None

    # Experiment specific
    violation_penalty: float = None
    scenario: str = None # "normal" / "vault" / "chimney"
    robot: str = None # "talos" / "g1"
    trajopt_type: str = None # "whole_body" / "centroidal"

    # Vault terrain
    vault_z: float = None
    vault_x_min: float = None
    vault_x_max: float = None
