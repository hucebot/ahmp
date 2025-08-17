import os
import time
import glob
import json

import numpy as np

from cem.params import Params
from cem.robot_trajopt import solve_trajopt
from robots.talos.TalosWrapper import Talos
from visualiser.visualiser import TrajoptVisualiser

from examples.cem_exps.exp_params import CHIMNEY_PARAMS, HANDRAILS_PARAMS

import imageio

file = "../laptop_results/handrails/run_1" + "/cem_solution.json"
# file = "../laptop_results/chimney/climb_low/run_5" + "/cem_solution.json"
# file = "../laptop_results/chimney/climb_high/run_3" + "/cem_solution.json"
with open(file, "r") as f:
    data = json.load(f)
    xd = np.array(data["solution"]["best_discrete"])
    xc = np.array(data["solution"]["best_continuous"])
p = HANDRAILS_PARAMS
# p = CHIMNEY_PARAMS
p.base_target_dz = 1.0
# p.base_target_dz = 3.0
p.print_level = 5
p.robot = "talos"

input = [xd.tolist(), np.exp(xc).tolist(), p]
result = solve_trajopt(input)

robot = Talos()

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
state_trajectory = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

tvis = TrajoptVisualiser(robot)

interp_states = state_trajectory
tvis.display_robot_q(robot, state_trajectory[0])

# Visualise the terrain
if p.scenario == "chimney":
    tvis.load_chimney_walls(y=0.55)
elif p.scenario == "handrails":
    tvis.load_handrails(rail_y=0.5, rail_z=0.9, rail_w=0.02)

# time.sleep(10)
# while True:
#     # Play motion plan forward and capture screenshots
#     for i in range(len(interp_states)):
#         time.sleep(dts[i])
#         tvis.display_robot_q(robot, interp_states[i])
#         tvis.update_forces(robot, forces[i], 0.01)
#         img = tvis.vis.captureImage()
#         imageio.imwrite(f"image_{i}.png", img)
#     # Additional frames for final configuration
#     for i in range(20):
#         time.sleep(0.05)
#         tvis.display_robot_q(robot, interp_states[-1])
#         tvis.update_forces(robot, {}, 0.01)
#         img = tvis.vis.captureImage()
#         imageio.imwrite(f"image_{1000+i}.png", img)
#
#     tvis.update_forces(robot, {}, 0.01)
#     time.sleep(5)

# Load multiple configurations to create snapshot images
print(len(interp_states))
# tvis.display_multiple_instances(robot, interp_states[0])
# tvis.display_multiple_instances(robot, interp_states[12])
# tvis.display_multiple_instances(robot, interp_states[35])
# tvis.display_multiple_instances(robot, interp_states[50])
# tvis.display_multiple_instances(robot, interp_states[71])

# while True:
#     pass
