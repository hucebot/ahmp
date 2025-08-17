import os
import time
from datetime import datetime
import argparse
import json

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cem.algo import CrossEntropyMethodMixed
from cem.params import Params
from cem.robot_trajopt import solve_trajopt

from robots.talos.TalosWrapper import Talos
from robots.g1.G1Wrapper import G1
from robots.go2.Go2Wrapper import Go2

from visualiser.visualiser import TrajoptVisualiser

from exp_params import CHIMNEY_PARAMS, HANDRAILS_PARAMS

parser = argparse.ArgumentParser()
parser.add_argument("--exp", help="Experiment to run ('vault', 'chimney')")
parser.add_argument("--robot", help="Robot name ('talos', 'g1'")
parser.add_argument("--min_time", help="Minimize time in CEM loop")
parser.add_argument("--viz", help="Whether to visualize the best results")
parser.add_argument("--dz", help="Target height for chimney scenario")
parser.add_argument("--abl", help="Elite percentage for ablation study")

args = parser.parse_args()

params = None

if args.exp == "chimney":
    params = CHIMNEY_PARAMS
    params.base_target_dz = float(args.dz)
    if args.abl:
        params.n_elites = int(float(args.abl) * params.pop_size)
    print(params.n_elites)
    print(params.base_target_dz)
elif args.exp == "handrails":
    params = HANDRAILS_PARAMS
    print(params.n_elites)
else:
    print("Error! Wrong experiment name")
    exit()

params.robot = args.robot
cost_hist = np.zeros(params.cem_iters)

algo = CrossEntropyMethodMixed(params)

prev_best = -np.inf
start = time.time()
for k in range(params.cem_iters):
    print("CEM-MD iteration: ", k)
    algo.generate_population_discrete()
    algo.generate_population_continuous()
    xd = algo.population_discrete

    xc = algo.population_continuous

    inputs = [
        [xd[:, i].tolist(), np.exp(xc[:, i]).tolist(), params]
        for i in range(params.pop_size)
    ]

    with ProcessPoolExecutor(max_workers=params.n_threads) as executor:
        results = list(executor.map(solve_trajopt, inputs))

    cost = np.zeros(params.pop_size)
    if args.min_time:
        cost += -np.sum(np.exp(xc[1:]), axis=0)

    n_viols = np.array([results[i]["n_viol"] for i in range(params.pop_size)])
    cost += params.viol_pen * n_viols

    algo.evaluate_population(cost)
    algo.update_distributions()

    cost_hist[k] = algo.log.best_value

    # Early exit criterion
    if not args.min_time:
        if algo.log.best_value == 0:
            break

end = time.time()

xd = np.copy(algo.log.best_discrete)
xc = np.copy(algo.log.best_continuous)

input = [xd.tolist(), np.exp(xc).tolist(), params]
result = solve_trajopt(input)

wall_time = end - start

report = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "scenario": params.scenario,
        "robot": params.robot,
        "cem_iterations": k + 1,
    },
    "solution": {
        "elite_cost_history": cost_hist.tolist(),
        "best_discrete": xd.tolist(),
        "best_continuous": xc.tolist(),
        "wall_time_sec": wall_time,
        "solution_found": result['info']['status'],
    },
}

# Save to file
filename = f"cem_solution.json"
save_path = os.path.join(os.path.abspath(os.getcwd()), filename)

with open(save_path, "w") as f:
    json.dump(report, f, indent=2)

if args.viz:
    robot = None
    if params.robot == "talos":
        robot = Talos()
    elif params.robot == "g1":
        robot = G1()
    elif params.robot == "go2":
        robot = Go2()
    else:
        print("Error, robot type not supported")
        exit()

    K = len(result["nodes"])
    dts = [result["nodes"][k]["dt"] for k in range(K)]
    state_trajectory = [result["nodes"][k]["q"] for k in range(K)]
    forces = [result["nodes"][k]["forces"] for k in range(K)]

    tvis = TrajoptVisualiser(robot)

    interp_states = state_trajectory
    tvis.display_robot_q(robot, state_trajectory[0])

    # Visualise the terrain
    if params.scenario == "vault":
        tvis.load_vault_obstacle(params.vault_z, params.vault_x_min, params.vault_x_max)
    elif params.scenario == "chimney":
        tvis.load_chimney_walls(params.chimney_y)
    # elif params.scenario == "handrails":
    #     tvis.load_handrails(params.chimney_y)

    time.sleep(1)
    while True:
        for i in range(len(interp_states)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, interp_states[i])
            tvis.update_forces(robot, forces[i], 0.01)

        tvis.update_forces(robot, {}, 0.01)
        time.sleep(3)
